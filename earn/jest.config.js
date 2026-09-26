'use strict';

// The gate is 100% across the board for every file listed below. One source
// file is not listed yet: src/main/pearlEngine.js (about 86% of its branches
// are covered today), so adding it is its own piece of work.
//
// main.js used to carry a per-file ratchet just below 100%: its uncovered
// branches were defensive fallbacks in the local-LLM and node-linking paths
// that were unreachable by construction. Those paths went with the LLM, and
// what is left is fully covered, so main.js is back under the global gate.
//
// Don't lower these numbers to make a red build green. If something genuinely
// cannot be reached, say why in the test file and ignore that one line with an
// istanbul hint, rather than loosening the gate for the whole file.
module.exports = {
  rootDir: '.',
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: [
    'src/shared/**/*.js',
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
  },
  testMatch: ['<rootDir>/test/**/*.test.js'],
};
