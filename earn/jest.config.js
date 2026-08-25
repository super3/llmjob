'use strict';

// Every source file is measured — and as of the llmFleet entry below, that is
// now actually true. The global gate is 100%; main.js alone has a per-file
// ratchet just below it — its uncovered branches are defensive fallbacks that
// are unreachable by construction (each documented in test/mainProcess.test.js),
// so 100% there would require deleting guards to satisfy a metric.
//
// Don't lower these numbers to make a red build green. The one legitimate reason
// to move `statements` DOWN is that main.js lost COVERED lines: the fixed set of
// unreachable statements then becomes a larger share of a smaller file. That is
// what happened when GPU detection moved out to probe.js (a ~10-line covered
// block replaced by a 4-line delegate), taking statements from 99.8 to 99.67.
// `branches` moved the other way in the same change and has been raised to match.
//
// It happened again, and much more so, when alpha-miner was removed. main.js
// lost its whole engine half -- resolving a version, checking the driver,
// finding a bundled binary, downloading one, unzipping it, spawning it and
// reporting the ways that fails -- about 180 lines, all of them covered. The
// unreachable set did not grow; it is the same defensive fallbacks it always
// was, now measured against a much smaller file. Statements 99.6 -> 98.6,
// branches 98.1 -> 97.0, functions 100 -> 98.5 (the two callbacks reached only
// by a real socket).
module.exports = {
  rootDir: '.',
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: [
    'src/shared/**/*.js',
    'src/main/minerManager.js', 'src/main/engineManager.js',
    'src/main/llmManager.js', 'src/main/llmEngineManager.js', 'src/main/jobWorker.js',
    // llmFleet was the one src/main file missing from this list while the header
    // above claimed every source file is measured. It is fully live — both shells
    // construct it — and the gap is not academic: two lifecycle bugs (an orphaned
    // llama-server spawned when stop() raced the port probe, and a reused fleet
    // that could never re-emit 'stopped') sat in the one file the 100% gate never
    // applied to.
    'src/main/llmFleet.js',
    'src/main/io.js', 'src/main/nodeStore.js', 'src/main/probe.js', 'src/main/preload.js',
    // Our own Pearl miner: the protocol/lifecycle half. The CUDA core it
    // drives lives in earn/native and is not measurable here — which is
    // exactly why the JS side is held to the full gate.
    'src/main/pearlMiner.js', 'src/main/pearlCore.js',
    'src/main/main.js', 'src/renderer/renderer.js',
    'src/cli/selfUpdater.js', 'src/cli/sea-entry.js', 'src/cli/earn-cli.js',
  ],
  coverageThreshold: {
    global: { branches: 100, functions: 100, lines: 100, statements: 100 },
    // Per-path thresholds pull main.js out of the global group (jest semantics),
    // so the 100% gate stays intact for everything else.
    'src/main/main.js': { branches: 97, functions: 98.5, lines: 100, statements: 98.6 },
  },
  testMatch: ['<rootDir>/test/**/*.test.js'],
};
