'use strict';

// Every source file is measured. The global gate is 100%; main.js alone has a
// per-file ratchet just below it — its seven uncovered branches are defensive
// fallbacks that are unreachable by construction (each documented in
// test/mainProcess.test.js), so 100% there would require deleting guards to
// satisfy a metric. Don't lower these numbers; raise them if main.js shrinks.
module.exports = {
  rootDir: '.',
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: [
    'src/shared/**/*.js',
    'src/main/minerManager.js', 'src/main/engineManager.js',
    'src/main/llmManager.js', 'src/main/llmEngineManager.js', 'src/main/jobWorker.js',
    'src/main/io.js', 'src/main/nodeStore.js', 'src/main/probe.js', 'src/main/preload.js',
    'src/main/main.js', 'src/renderer/renderer.js',
    'src/cli/selfUpdater.js', 'src/cli/sea-entry.js', 'src/cli/earn-cli.js',
  ],
  coverageThreshold: {
    global: { branches: 100, functions: 100, lines: 100, statements: 100 },
    // Per-path thresholds pull main.js out of the global group (jest semantics),
    // so the 100% gate stays intact for everything else.
    'src/main/main.js': { branches: 97.9, functions: 100, lines: 100, statements: 99.8 },
  },
  testMatch: ['<rootDir>/test/**/*.test.js'],
};
