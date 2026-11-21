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
//
// The app starts again only on a rig with more cards than the default pool has
// threads. Everyone else starts as before, with no second process. The count
// comes from the last run (CARD_COUNT_FILE), because asking nvidia-smi here
// would slow every start.
const POOL_THREADS = 32;
const DEFAULT_THREADS = 4;

// Where the app keeps the number of GPUs it last found (main.js, detectGpu):
// in its userData folder, which on Windows is %APPDATA%\<package name>. Read
// here before Electron loads, so the folder is built by hand.
const APP_DIR = 'llmjob-earn';
const CARD_COUNT_FILE = 'gpu-count.json';

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

// The number of GPUs the app found last run, or null when it never saved one
// (the first run after install) or this is not Windows, where nothing needs it.
function savedCardCount(proc, fs, join) {
  if (proc.platform !== 'win32' || !proc.env.APPDATA) return null;
  try {
    const n = JSON.parse(fs.readFileSync(join(proc.env.APPDATA, APP_DIR, CARD_COUNT_FILE), 'utf8')).cards;
    return Number.isInteger(n) && n >= 0 ? n : null;
  } catch (e) {
    return null;
  }
}

// The Electron app (main.js, its first line). On Windows without the variable,
// on a rig whose saved card count (`cards`, savedCardCount) is above
// DEFAULT_THREADS, start the app again with it and exit, before this copy
// takes the single-instance lock or opens a window. The new copy gets the
// same arguments (--hidden for a launch at login). A Windows rig with fewer
// cards, or no saved count, starts in this process as it did before the pool
// was widened, and the environment is left alone. Anywhere else, set it here,
// which libuv reads when the pool is first used. True when it started again;
// `proc.exit` doesn't return then, except in tests.
function relaunchApp(proc, spawn, cards) {
  if (!proc.versions.electron || !needsRelaunch(proc)) {
    widenThreadPool(proc.env);
    return false;
  }
  if (!(cards > DEFAULT_THREADS)) return false;
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

module.exports = {
  POOL_THREADS, DEFAULT_THREADS, APP_DIR, CARD_COUNT_FILE,
  widenThreadPool, needsRelaunch, relaunchEnv, savedCardCount, relaunchApp, relaunchCli,
};
