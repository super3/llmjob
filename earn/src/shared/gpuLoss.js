'use strict';

// Cards that kept losing their GPU under a model.
//
// LlmManager gives up on a card after it loses the GPU gpuLossLimit times in an
// hour (a driver reset, Vulkan's ErrorDeviceLost). That only lasted until the
// next start: start() forgets the losses, so every START and every launch at
// login loaded the model there again, at the cost of up to three more driver
// resets. So the shell remembers the card and the model, and the next start
// skips that model on that card.
//
// The record is { version, cards: { <card key>: [model name, …] } }. One from
// another app version counts as empty, since the update may have fixed it.

const { normalizePciBusId } = require('./gpu');

// A card's PCI bus id, which stays with it when others are added or moved:
// `busId` when the caller has one (the bus id llama-server loaded on), else
// nvidia-smi's from `busIds`. Null without either.
function busKey(index, busIds, busId) {
  return normalizePciBusId(busId || (busIds || {})[index]);
}

// A card's number and name. A null `index` is the card llama.cpp picked itself,
// on a rig nvidia-smi can't read.
function namedKey(index, cards) {
  if (index == null) return 'auto';
  const card = (cards || []).find((c) => c && c.index === index);
  return index + ':' + ((card && card.name) || '');
}

// Which keys a loss is saved under and looked up by. A start does not always
// know the bus ids: nvidia-smi can fail to give them on one start and not the
// next, and on a rig it can't read at all only llama-server's own bus id may be
// known. Saving under one key and looking up under another lost the record, and
// the model went back on the failing card. So a loss is saved under the bus id
// when known and always under the number and name, marked when a bus id was
// saved too. A start that knows the bus id trusts it, plus a number-and-name
// record saved without one; the marked one is left out, since numbers change
// when cards are moved. A start without bus ids reads both number-and-name keys.
const WITH_BUS = ' +bus';

function saveKeys(index, cards, busIds, busId) {
  const bus = busKey(index, busIds, busId);
  const named = namedKey(index, cards);
  return bus ? [bus, named + WITH_BUS] : [named];
}

function lookupKeys(index, cards, busIds) {
  const bus = busKey(index, busIds);
  const named = namedKey(index, cards);
  return bus ? [bus, named] : [named, named + WITH_BUS];
}

// The models that kept losing the GPU on a card, from its lookup keys.
function lostModels(record, keys) {
  const names = [];
  for (const key of keys) {
    for (const name of record.cards[key] || []) if (!names.includes(name)) names.push(name);
  }
  return names;
}

// The saved record if it is this version's, else an empty one. Anything that
// isn't a list of names is dropped, so a hand-edited file can't break a start.
function forVersion(saved, version) {
  const cards = {};
  if (saved && saved.version === version && saved.cards && typeof saved.cards === 'object') {
    for (const [key, names] of Object.entries(saved.cards)) {
      if (Array.isArray(names)) cards[key] = names.filter((n) => typeof n === 'string');
    }
  }
  return { version, cards };
}

// The record with `name` added for the card under each of its `keys`.
function remember(record, keys, name) {
  let cards = record.cards;
  for (const key of keys) {
    const had = cards[key] || [];
    if (!had.includes(name)) cards = Object.assign({}, cards, { [key]: had.concat(name) });
  }
  return cards === record.cards ? record : { version: record.version, cards };
}

module.exports = { saveKeys, lookupKeys, lostModels, forVersion, remember };
