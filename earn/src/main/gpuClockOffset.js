'use strict';

// Read and set a card's core clock offset, for the trial behind
// --mine-core-offset auto (shared/coreOffset says which cards, and
// shared/coreOffsetTrial how the trial goes). This file is only the call.
//
// nvidia-smi has no switch for an offset, so there is no command to run as
// main/gpuClocks does for the memory clock. The call goes to NVML, the driver's
// own library, through the mining core: the CUDA builds of pearl_core.node
// export getClockOffset(index) and setClockOffset(index, mhz) beside createCore
// (native/src/pearl_core.cc, "The core clock offset"). They open NVML when
// first called, so a core loads the same on a machine without it.
//
// So the offset calls belong to an addon, and main/pearlCore hands them to the
// miner with the createCore it built from that addon. A core with no such
// exports -- the AMD core, or a pearl_core.node built before them -- answers
// "not supported by this core".
//
// Setting an offset needs the rights nvidia-smi needs to set a clock, but there
// is no `sudo -n` to go through here: the call is made by this process, so the
// process itself must run as root, or on Windows as administrator. NVML says so
// with its code 4, which comes back as `noRights`.
//
// Nothing here throws; every call answers { ok, error } and the caller decides
// what to say.

const NOT_SUPPORTED = 'not supported by this core';
// NVML_ERROR_NO_PERMISSION, as the addon passes it on in `code`.
const NVML_NO_PERMISSION = 4;

// What went wrong, as { ok: false, error, noRights }. `r` is the addon's answer,
// or what it threw: the addon is written not to throw, but a core built by
// someone else is still only a file on disk.
function failure(r) {
  const o = Object(r);
  return {
    ok: false,
    error: String(o.error || o.message || 'the core gave no answer'),
    noRights: o.code === NVML_NO_PERMISSION,
  };
}

// Card `index`'s core clock offset and the range its driver allows, in MHz, as
// { ok: true, error: null, mhz, min, max }.
function readCoreOffset(addon, index) {
  if (!addon || typeof addon.getClockOffset !== 'function') return failure({ error: NOT_SUPPORTED });
  let r;
  try {
    r = addon.getClockOffset(index);
  } catch (e) {
    return failure(e);
  }
  if (!r || r.ok !== true) return failure(r);
  if (![r.mhz, r.min, r.max].every(Number.isInteger)) return failure({ error: 'the core gave no usable answer' });
  return { ok: true, error: null, mhz: r.mhz, min: r.min, max: r.max };
}

// Set card `index`'s core clock offset to `mhz`. 0 is the card's own clock.
function setCoreOffset(addon, index, mhz) {
  if (!addon || typeof addon.setClockOffset !== 'function') return failure({ error: NOT_SUPPORTED });
  let r;
  try {
    r = addon.setClockOffset(index, mhz);
  } catch (e) {
    return failure(e);
  }
  if (!r || r.ok !== true) return failure(r);
  return { ok: true, error: null };
}

// The two calls bound to an addon, as PearlMiner takes them:
// { readCoreOffset(index), setCoreOffset(index, mhz) }. `current` returns the
// addon to call, asked on every call: pearlCore's fallbacks can change which
// build a rig's cores come from after its first card opens.
function clockOffsets(current) {
  return {
    readCoreOffset: (index) => readCoreOffset(current(), index),
    setCoreOffset: (index, mhz) => setCoreOffset(current(), index, mhz),
  };
}

// For a miner whose createCore carries no offset calls.
const NO_OFFSETS = clockOffsets(() => null);

module.exports = { clockOffsets, NO_OFFSETS, NOT_SUPPORTED, NVML_NO_PERMISSION };
