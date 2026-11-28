'use strict';

// Entry point for the packaged single-file binary (Node SEA). A SEA has no
// `require.main === module`, so earn-cli.js's self-invoke guard never fires;
// this shim calls main() explicitly, which runs the CLI and exits with its code.
// A SEA's process.argv is [execPath, invokedPath, ...userArgs] — user args
// start at index 2, same as a normal `node script.js …` invocation, so
// slice(2) is correct in both.

//
// A SEA also takes no `-e`, so when the Quantus core checks a patched kernel in
// a child process (src/main/quantusCubinCheck.js), the child is this binary
// with QUANTUS_CUBIN_CHECK_FILE set: it runs that check and nothing else.

const { CHECK_FILE, runChild } = require('../main/quantusCubinCheck');

if (process.env[CHECK_FILE]) runChild(process);
else require('./earn-cli').main(process.argv.slice(2));
