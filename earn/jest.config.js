'use strict';

// Every source file is measured, and every one is held to the 100% gate.
//
// main.js had a per-file ratchet below it, and the parts split out of
// it (updater, appWindow, appLlm, appNode, appSession) briefly took entries of
// their own. The gaps were a few defensive fallbacks and callbacks no test
// drove. They now have tests, or were dropped where the caller already
// guarantees the case, so the exceptions are gone. Don't add one back to make
// a red build green: test the path, or simplify the code that cannot be reached.
module.exports = {
  rootDir: '.',
  testEnvironment: 'node',
  collectCoverage: true,
  collectCoverageFrom: [
    'src/shared/**/*.js',
    'src/main/llmManager.js', 'src/main/llmEngineManager.js', 'src/main/jobWorker.js',
    // llmFleet was the one src/main file missing from this list while the header
    // above claimed every source file is measured. It is fully live — both shells
    // construct it — and the gap is not academic: two lifecycle bugs (an orphaned
    // llama-server spawned when stop() raced the port probe, and a reused fleet
    // that could never re-emit 'stopped') sat in the one file the 100% gate never
    // applied to.
    'src/main/llmFleet.js',
    // Demand-driven auto mode. Same argument as llmFleet above: it is fully live
    // whenever a card can serve a model it cannot co-run, and the whole point of
    // it is lifecycle -- stopping one engine to start another -- which is exactly
    // the kind of code the 100% gate is worth having on.
    'src/main/llmGateServer.js', 'src/main/autoGate.js', 'src/main/lanShare.js',
    'src/main/io.js', 'src/main/nodeStore.js', 'src/main/probe.js', 'src/main/preload.js',
    // What both shells share for the network node and the LLM's start-up. Their
    // copies used to sit in main.js, outside this gate, and in the CLI.
    'src/main/nodeClient.js', 'src/main/llmLaunch.js',
    // Our own Pearl miner: the protocol/lifecycle half. The CUDA core it
    // drives lives in earn/native and is not measurable here — which is
    // exactly why the JS side is held to the full gate.
    'src/main/pearlMiner.js', 'src/main/pearlCore.js', 'src/main/pearlEngine.js',
    // The memory clock lock (the Blackwell default, or --mine-mem-clock).
    // Releasing the lock is what keeps a served LLM off a card with its memory
    // clock cut, so it is held to the gate like the miner is.
    'src/main/gpuClocks.js',
    // The core clock offset calls (--mine-core-offset auto). They run a card
    // above the clock its maker set, so what they answer when the addon has no
    // such calls, refuses or throws is held to the gate too.
    'src/main/gpuClockOffset.js',
    // The Monero CPU miner's host: the pool socket and the worker process it
    // drives. The worker itself is C++ (earn/xmr-worker) and checks itself
    // against RandomX's test vectors, so the JS side is held to the gate.
    'src/main/xmrEngine.js', 'src/main/xmrWorker.js', 'src/main/cpuFacts.js', 'src/main/logFile.js',
    // main.js and the parts split out of it, along its section markers.
    'src/main/main.js', 'src/main/balances.js', 'src/main/cpuMining.js', 'src/main/appContext.js',
    'src/main/updater.js', 'src/main/appWindow.js', 'src/main/appLlm.js', 'src/main/appNode.js',
    'src/main/appSession.js',
    'src/renderer/renderer.js',
    'src/cli/selfUpdater.js', 'src/cli/sea-entry.js', 'src/cli/earn-cli.js',
  ],
  coverageThreshold: {
    global: { branches: 100, functions: 100, lines: 100, statements: 100 },
  },
  testMatch: ['<rootDir>/test/**/*.test.js'],
};
