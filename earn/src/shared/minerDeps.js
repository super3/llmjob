'use strict';

// What both shells build their miners from: the Pearl engine, the Monero engine,
// the Quantus engine (the CLI's --coin quantus), the pool-name lookup they
// share, and the two small caches a Pearl start reads. main.js and earn-cli.js
// each wired these by hand before.
//
// `version`    this client's version; the pool login's agent string is
//              'llmjob-earn/' + version, for every coin.
// `nodeDir()`  the folder node.json is in (nodeStore.nodePath()). The pool
//              cache (pool-addresses.json) is kept there by both shells, so the
//              desktop app and the CLI on one machine share it.
// `storeDir()` where the launch shapes (launch-shapes.json), the k trial
//              answers (k-trials.json) and the memory clock trial answers
//              (mem-clock-trials.json) are kept. The two
//              shells differ here: the desktop app keeps them in its userData
//              folder, the CLI beside node.json. That is where each has always
//              kept them, and moving them would run every card's trials again
//              on users' PCs, so each shell passes its own folder. Both are
//              asked when a file is first needed, not when this is made.
// `fs`         the fs module (injected so the shells' tests see the writes).
// `settingsLog` hears a cache file that cannot be read or written.
// `lookupLog`  hears each fallback the pool lookup uses.

const net = require('net');
const tls = require('tls');
const dns = require('dns');
const path = require('path');
const { PearlEngine } = require('../main/pearlEngine');
const { XmrEngine } = require('../main/xmrEngine');
const { QuantusEngine } = require('../main/quantusEngine');
const probe = require('../main/probe');
const { makePoolLookup, fileCache } = require('./poolLookup');
const { shapeFileStore } = require('./shapePick');
const { readSettings, writeSettings } = require('./settingsStore');

function minerDeps({ version, nodeDir, storeDir, fs, settingsLog, lookupLog }) {
  const agent = 'llmjob-earn/' + version;

  // How the pool's name becomes an address: the system lookup, then fallbacks
  // when it fails (shared/poolLookup). Made once, for both miners: the cache is
  // one file, and two copies of it would overwrite each other.
  let lookupFn = null;
  const poolLookup = () => {
    if (!lookupFn) {
      lookupFn = makePoolLookup({ dns, cache: fileCache(path.join(nodeDir(), 'pool-addresses.json'), fs, path), log: lookupLog });
    }
    return lookupFn;
  };
  // `secure`: the pool address was typed ssl://, so the pool's TLS port. The
  // socket verifies the pool's certificate against its name, and the engines
  // use it as they do a plain one: what they write waits for the handshake.
  const connect = (host, port, secure) => (secure ? tls : net).connect({ port, host, lookup: poolLookup() });
  // The pool took the login, so this address is one that worked.
  const rememberAddress = (engine) => engine.on('authorized', (e) => poolLookup().worked(e.host, e.address, e.family));
  const store = (name) => shapeFileStore(path.join(storeDir(), name), { readSettings, writeSettings, fs, log: settingsLog });

  return {
    agent,
    poolLookup,
    // Each RTX 30 card's launch shape, timed once on this PC (shared/shapePick).
    shapeStore: () => store('launch-shapes.json'),
    // The Pearl miner. `opts` has its createCore. The card temperature comes
    // from nvidia-smi, as the core has no reading of its own, and each card's
    // memory clock trial answer is kept for a week, so a card is not tried
    // again on every start (shared/memClockTrial).
    pearlEngine: (opts) => {
      const engine = new PearlEngine(Object.assign({
        connect,
        readTemps: () => probe.detectGpuTemps(),
        agent,
        memClockStore: store('mem-clock-trials.json'),
        // And each RX 9070 XT's k, k 8192 or k 16384 (shared/kTrial).
        kStore: store('k-trials.json'),
      }, opts));
      rememberAddress(engine);
      return engine;
    },
    // The Monero miner. `opts` has its spawnWorker and workerPath.
    xmrEngine: (opts) => {
      const engine = new XmrEngine(Object.assign({ connect, agent }, opts));
      rememberAddress(engine);
      return engine;
    },
    // The Quantus miner. `opts` has its core factory (main/quantusCore).
    quantusEngine: (opts) => {
      const engine = new QuantusEngine(Object.assign({ connect, agent }, opts));
      rememberAddress(engine);
      return engine;
    },
  };
}

module.exports = { minerDeps };
