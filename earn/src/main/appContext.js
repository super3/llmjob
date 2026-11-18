'use strict';

// What the parts of the desktop app's main process share.
//
// main.js was one file whose sections read and wrote each other's variables:
// the window, the miner, the LLM fleet, the user's START. Split into modules,
// each part keeps its own state and puts on `ctx` what the others need:
//   - a live variable, as a getter, plus a setter where another part writes it
//   - the functions the others call
// Nothing is copied. A part reads `ctx.miner` or calls `late('send')` at the
// time of use, so it always sees the current value, and the parts can load in
// any order.

const ctx = {};

// `vars`: { name: { get, set } }, one getter (and setter) per live variable.
// `fns`: { name: fn }, the functions the other parts call.
function share(vars, fns) {
  for (const name of Object.keys(vars)) {
    Object.defineProperty(ctx, name, Object.assign({ enumerable: true, configurable: true }, vars[name]));
  }
  Object.assign(ctx, fns);
}

// A function another part shares, looked up when it is called.
function late(name) {
  return (...args) => ctx[name](...args);
}

module.exports = { ctx, share, late };
