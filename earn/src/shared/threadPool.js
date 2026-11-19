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
const POOL_THREADS = 32;

// Sets UV_THREADPOOL_SIZE in `env` to POOL_THREADS when nobody chose a size.
// True when it set it.
function widenThreadPool(env) {
  if (String(env.UV_THREADPOOL_SIZE || '').trim()) return false;
  env.UV_THREADPOOL_SIZE = String(POOL_THREADS);
  return true;
}

module.exports = { POOL_THREADS, widenThreadPool };
