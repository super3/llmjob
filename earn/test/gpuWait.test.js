'use strict';

const fs = require('fs');
const path = require('path');

// How a card's search thread waits for the GPU between launches (PEARL_WAIT in
// native/src/pearl_host.cu). Up to v0.6.15 it busy-waited, a whole CPU core a card, and a
// rig of 8 cards on a 2-core CPU ran at under half its rate. Nothing here can run the
// device code, so these pin the source: the default sleeps, the three choices are the
// ones the CLI offers, and only a thread that sleeps raises its own priority.

const NL = String.fromCharCode(10);
const read = (...parts) => fs.readFileSync(path.join(__dirname, '..', ...parts), 'utf8')
  .split(String.fromCharCode(13) + NL).join(NL);
const HOST = read('native', 'src', 'pearl_host.cu');
const { parseCliArgs } = require('../src/shared/cliArgs');

// The body of a function, from its signature to the first line that is only "}".
function body(signature) {
  const at = HOST.indexOf(signature);
  expect(at).toBeGreaterThan(-1);
  return HOST.slice(at, HOST.indexOf(NL + '}' + NL, at));
}

describe('the CUDA core\'s wait', () => {
  test('sleeps by default', () => {
    expect(HOST).toContain('#define PEARL_WAIT_DEFAULT PEARL_WAIT_BLOCK');
    expect(HOST).toContain('int waitMode = PEARL_WAIT_DEFAULT;');
  });

  test('PEARL_WAIT takes the choices --gpu-wait offers, and no others', () => {
    const fn = body('int pearl_wait_mode() {');
    const named = [...fn.matchAll(/strcmp\(w, "([a-z]+)"\)/g)].map((m) => m[1]).sort();
    expect(named).toEqual(['block', 'poll', 'spin']);
    for (const w of named) expect(parseCliArgs(['-a', 'x', '--gpu-wait', w]).settings.gpuWait).toBe(w);
    // Anything else keeps the default and says so.
    expect(fn).toContain('ignored: it takes block, poll or spin');
  });

  test('a blocking wait is a blocking event, and a poll never calls the busy wait', () => {
    expect(HOST).toContain('(ctx->waitMode == PEARL_WAIT_BLOCK ? cudaEventBlockingSync : 0)');
    const fn = body('extern "C" bool pearl_host_collect(');
    const poll = fn.indexOf('if (ctx->waitMode == PEARL_WAIT_POLL) {');
    const other = fn.indexOf('} else {', poll);
    expect(poll).toBeGreaterThan(-1);
    expect(other).toBeGreaterThan(poll);
    expect(fn.slice(poll, other)).toContain('cudaEventQuery(ctx->slotDone[slot])');
    expect(fn.slice(poll, other)).toContain('pearl_nap(ctx, PEARL_WAIT_POLL_US)');
    expect(fn.slice(poll, other)).not.toContain('cudaEventSynchronize');
    // One busy-or-blocking wait in the whole function, in the other branch.
    expect(fn.split('cudaEventSynchronize(').length - 1).toBe(1);
  });

  test('only a search thread that sleeps raises its priority', () => {
    const fn = body('extern "C" void pearl_host_bind_thread(');
    const guard = fn.indexOf('if (ctx->waitMode != PEARL_WAIT_SPIN) {');
    expect(guard).toBeGreaterThan(-1);
    expect(fn.indexOf('SetThreadPriority(')).toBeGreaterThan(guard);
    expect(fn.indexOf('setpriority(')).toBeGreaterThan(guard);
  });
});
