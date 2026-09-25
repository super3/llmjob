'use strict';

// Lock and release a card's memory clock while it mines (--mine-mem-clock).
//
// Why a miner would want LESS memory clock: on a power-capped card the fold is
// power-bound, not bandwidth-bound. Nsight Compute on an RTX 5090 puts it at
// ~5 GB/s of DRAM traffic -- 418 MB per launch at a 99.6% L2 hit rate -- so
// GDDR7 at its default 13801 MHz mostly burns watts the SMs could have used.
// Locked at 7001 MHz under the same 600 W cap, the SM clock rose from 682 to
// 742 MHz and the fold from 95.58 / 95.90 to 103.18 / 103.55 TH/s (+8.0%, same
// core, interleaved runs), with the same work per SM clock. Lower still (810 or
// 405 MHz) raises the SM clock further but the work per clock drops -- the
// L2/crossbar appears to slow with the memory P-state -- so 7001 is the value
// to use on a 5090.
//
// LLM decode is the opposite: llama-server is memory-bandwidth-bound, so a card
// must never serve a model with this lock in place. The miner releases it in
// stop(), synchronously, which is why this uses execFileSync: the reset has
// finished before anything that follows a stop -- the demand gate starting
// llama-server, or the process exiting -- gets to run.
//
// Setting clocks needs root. A non-root process goes through `sudo -n`, which
// never prompts: with no NOPASSWD rule it fails at once, and the miner carries
// on at the default clock. Nothing here throws; every call answers
// { ok, error } and the caller decides what to say.

const { execFileSync } = require('child_process');

// An nvidia-smi call answers in well under a second. Five is the same budget
// probe.js gives its queries, and it bounds how long a wedged driver or sudo
// can hold up a start or a stop.
const TIMEOUT_MS = 5000;

// On a platform with no uids (Windows) there is no sudo to go through either.
function currentUid() {
  return process.getuid ? process.getuid() : 0;
}

// What went wrong, in one line. sudo complains on stderr and nvidia-smi
// usually on stdout; a spawn failure (no sudo, no nvidia-smi, the timeout)
// has neither and only a message. Object() so that whatever was thrown can be
// read without a guard.
function reason(e) {
  const { stderr, stdout, message } = Object(e);
  const said = [stderr, stdout].map((s) => String(s || '').trim()).find(Boolean);
  return said ? said.split(/\r?\n/)[0] : String(message || e);
}

// Run nvidia-smi with `args`, as root or through `sudo -n`. argv, not a shell
// string: nothing here is ever interpolated into a command line.
function nvidiaSmi(args, { exec = execFileSync, getuid = currentUid } = {}) {
  const [cmd, argv] = getuid() === 0
    ? ['nvidia-smi', args]
    : ['sudo', ['-n', 'nvidia-smi'].concat(args)];
  try {
    exec(cmd, argv, { timeout: TIMEOUT_MS, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    return { ok: true, error: null };
  } catch (e) {
    return { ok: false, error: reason(e) };
  }
}

// Pin card `index`'s memory clock to exactly `mhz` (min and max the same).
function lockMemoryClock(index, mhz, deps) {
  return nvidiaSmi(['-i', String(index), '-lmc', mhz + ',' + mhz], deps);
}

// Hand card `index`'s memory clock back to the driver.
function resetMemoryClock(index, deps) {
  return nvidiaSmi(['-i', String(index), '-rmc'], deps);
}

module.exports = { lockMemoryClock, resetMemoryClock, TIMEOUT_MS };
