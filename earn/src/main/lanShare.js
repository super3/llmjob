'use strict';

// Share the desktop app's local LLM with other devices on the network (#257).
//
// The app's llama-server listens on 127.0.0.1 only, so nothing else on the
// network can reach it. With sharing on, the app runs the CLI's serve gate
// (autoGate.createServeGate) in front of it: the same proxy, on the same port
// 8000, that the CLI serves on in llm and auto modes, bound to every interface.
// llama-server itself stays on loopback, so the in-app chat and the cluster
// worker don't change.
//
// The gate runs only while the switch is on AND a model is answering. The app
// calls sync() whenever either of those changes.

const os = require('os');
const { createServeGate } = require('./autoGate');
const { LLM } = require('../shared/config');

// Home networks first, so the address shown is the one a phone on the same
// Wi-Fi can use, not a VPN's or a virtual switch's (WSL, Hyper-V, Docker).
function rank(addr) {
  if (addr.startsWith('192.168.')) return 0;
  if (addr.startsWith('10.')) return 1;
  if (/^172\.(1[6-9]|2\d|3[01])\./.test(addr)) return 2;
  return 3;
}

// The IPv4 addresses other devices can reach this computer on: every interface
// except loopback and link-local (169.254.x.x, what Windows assigns when DHCP
// fails, which nothing else can route to).
function lanAddresses(ifaces) {
  const out = [];
  for (const list of Object.values(ifaces || {})) {
    for (const a of list || []) {
      if (a.family === 'IPv4' && !a.internal && !a.address.startsWith('169.254.')) out.push(a.address);
    }
  }
  return out.sort((x, y) => rank(x) - rank(y));
}

function createLanShare(opts = {}) {
  const {
    upstreamUrl, isLlmReady, log = () => {}, onChange = () => {},
    port = LLM.gate.port, interfaces = os.networkInterfaces, makeGate = createServeGate,
  } = opts;
  let gate = null;
  let error = null;

  const urls = () => lanAddresses(interfaces()).map((a) => 'http://' + a + ':' + port + '/v1');

  function stop() {
    gate.stop();
    gate = null;
  }

  // want: whether the gate should be up now. model: { name, ctxSize } of the
  // model being served, for the gate's answers to /v1/models and /props.
  function sync(want, model = {}) {
    if (want && !gate) {
      error = null;
      const g = makeGate({
        port,
        // NOT LLM.port: the fleet walks upward from it when 8080 is taken, so
        // ask it where it actually bound, as the CLI does.
        upstreamPort: () => {
          const url = upstreamUrl();
          const p = url ? Number(new URL(url).port) : NaN;
          return Number.isFinite(p) && p > 0 ? p : LLM.port;
        },
        modelName: model.name,
        ctxSize: model.ctxSize,
        isLlmReady,
        log,
        // Port 8000 taken (often by the CLI on the same computer): say so on the
        // switch, not only in the log, and let the next sync try again. Ignored
        // if this gate was already stopped and replaced.
        onListenError: (e) => {
          if (gate !== g) return;
          error = e.code === 'EADDRINUSE' ? 'Port ' + port + ' is already in use' : e.message;
          stop();
          onChange();
        },
      });
      gate = g;
      g.start();
      const list = urls();
      log('sharing the local LLM on your network at '
        + (list.length ? list.join(', ') : 'port ' + port + ' (no network address yet)'));
      onChange();
    } else if (!want && (gate || error)) {
      if (gate) {
        stop();
        log('stopped sharing the local LLM on the network');
      }
      error = null;
      onChange();
    }
  }

  // What the app shows: null while not shared, else the addresses to use and
  // any error.
  function status() {
    if (error) return { urls: [], error };
    if (!gate) return null;
    return { urls: urls(), error: null };
  }

  return { sync, status };
}

module.exports = { createLanShare, lanAddresses };
