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

// One card, the same way on every start: its PCI bus id, which stays with the
// card when others are added or moved, else its number and name. `busId` is a
// bus id the caller already has for it (the one llama-server loaded on), else
// nvidia-smi's from `busIds`. A null `index` is the card llama.cpp picked
// itself, on a rig nvidia-smi can't read.
function cardKey(index, cards, busIds, busId) {
  const bus = normalizePciBusId(busId || (busIds || {})[index]);
  if (bus) return bus;
  if (index == null) return 'auto';
  const card = (cards || []).find((c) => c && c.index === index);
  return index + ':' + ((card && card.name) || '');
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

// The record with `name` added for the card `key`.
function remember(record, key, name) {
  const had = record.cards[key] || [];
  if (had.includes(name)) return record;
  return { version: record.version, cards: Object.assign({}, record.cards, { [key]: had.concat(name) }) };
}

module.exports = { cardKey, forVersion, remember };
