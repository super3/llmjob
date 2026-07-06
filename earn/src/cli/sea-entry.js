'use strict';

// Entry point for the packaged single-file binary (Node SEA). A SEA has no
// `require.main === module`, so earn-cli.js's self-invoke guard never fires;
// this shim calls run() explicitly. A SEA's process.argv is
// [execPath, invokedPath, ...userArgs] — user args start at index 2, same as a
// normal `node script.js …` invocation, so slice(2) is correct in both.

const { run } = require('./earn-cli');

run(process.argv.slice(2))
  .then((code) => { process.exitCode = code; })
  .catch((e) => { process.stderr.write('fatal: ' + (e && e.message ? e.message : e) + '\n'); process.exitCode = 1; });
