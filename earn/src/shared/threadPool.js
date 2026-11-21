'use strict';

// libuv's thread pool, which N-API async work runs on. It has 4 threads unless
// UV_THREADPOOL_SIZE says otherwise, and libuv reads that once, when the pool is
// first used, so the shells set it before anything else runs.
//
// Each Quantus core (earn/native/quantus: CUDA, HIP and SYCL alike) runs a scan
// as async work, and the scan holds its pool thread while the card works, about
// 100 ms a batch (main/quantusEngine.js). With 4 threads, a rig with 6 cards
// kept only 4 busy, and the batch sizing read the others' wait as a slow card.
// The Pearl core runs a thread of its own a card, so Pearl never needed this.
// DNS lookups and async fs share the pool too. 32 covers the largest rigs.
//
// On Windows, setting the variable from inside the process does nothing: by the
// time a script runs, libuv there no longer reads it. Checked on Node 22 and
// Electron 33 with 12 pbkdf2 jobs (scripts/check-thread-pool.js): set in the
// process they still ran 4 at a time, set before the start 12 at once. So on
// Windows a shell started without it starts itself again with it set
// (relaunchApp, relaunchCli). HiveOS's h-run.sh sets it before the CLI starts.
const POOL_THREADS = 32;

const chosen = (env) => !!String(env.UV_THREADPOOL_SIZE || '').trim();

// Sets UV_THREADPOOL_SIZE in `env` to POOL_THREADS when nobody chose a size.
// True when it set it.
function widenThreadPool(env) {
  if (chosen(env)) return false;
  env.UV_THREADPOOL_SIZE = String(POOL_THREADS);
  return true;
}

// Whether this process must start again to get the pool: on Windows, when
// nobody chose a size.
function needsRelaunch(proc) {
  return proc.platform === 'win32' && !chosen(proc.env);
}

// `env` with the pool set, for the process started again.
function relaunchEnv(env) {
  return Object.assign({}, env, { UV_THREADPOOL_SIZE: String(POOL_THREADS) });
}

// The Electron app (main.js, its first line). On Windows without the variable,
// start the app again with it and exit, before this copy takes the
// single-instance lock or opens a window. The new copy gets the same
// arguments (--hidden for a launch at login). Anywhere else, set it here,
// which libuv reads when the pool is first used. True when it started again;
// `proc.exit` doesn't return then, except in tests.
function relaunchApp(proc, spawn) {
  if (!proc.versions.electron || !needsRelaunch(proc)) {
    widenThreadPool(proc.env);
    return false;
  }
  try {
    spawn(proc.execPath, proc.argv.slice(1), { env: relaunchEnv(proc.env), detached: true, stdio: 'ignore' }).unref();
  } catch (e) {
    // Mining on 4 pool threads beats not starting at all.
    widenThreadPool(proc.env);
    return false;
  }
  proc.exit(0);
  return true;
}

// The CLI run with node on Windows (the release's CLI is Linux only), when
// needsRelaunch said so before anything set the variable. Runs it again as a
// child with the variable set, in this console, and resolves with its exit
// code, or with null when the child could not start: the caller then runs in
// this process. Ctrl+C reaches the child from the console itself, so this copy
// ignores it and waits for the child's exit.
function relaunchCli(proc, spawn) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(proc.execPath, proc.execArgv.concat(proc.argv.slice(1)), { env: relaunchEnv(proc.env), stdio: 'inherit' });
    } catch (e) {
      resolve(null);
      return;
    }
    const ignore = () => {};
    proc.on('SIGINT', ignore);
    proc.on('SIGTERM', ignore);
    child.on('error', () => resolve(null));
    child.on('exit', (code) => resolve(code == null ? 1 : code));
  });
}

module.exports = { POOL_THREADS, widenThreadPool, needsRelaunch, relaunchEnv, relaunchApp, relaunchCli };
