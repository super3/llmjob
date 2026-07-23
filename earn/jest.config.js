'use strict';

// Coverage is enforced at 100% on the pure-logic modules (shared/*), the
// process supervisors (managers), the IO helpers, and the small Electron/CLI
// bootstraps (preload.js, sea-entry.js). The remaining entry points that need a
// live Electron/DOM/CLI runtime — main.js, renderer.js, earn-cli.js — are
// excluded; covering them needs an end-to-end harness, not unit tests.
module.exports = {
  rootDir: '.',
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: [
    'src/shared/**/*.js',
    'src/main/minerManager.js', 'src/main/engineManager.js',
    'src/main/llmManager.js', 'src/main/llmEngineManager.js', 'src/main/jobWorker.js',
    'src/main/io.js', 'src/main/nodeStore.js', 'src/main/probe.js', 'src/main/preload.js',
    'src/cli/selfUpdater.js', 'src/cli/sea-entry.js',
  ],
  coverageThreshold: {
    global: { branches: 100, functions: 100, lines: 100, statements: 100 },
  },
  testMatch: ['<rootDir>/test/**/*.test.js'],
};
