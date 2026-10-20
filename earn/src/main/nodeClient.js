'use strict';

// This machine's side of the LLMJob network, for both shells: register, ping,
// join, telemetry, and the cluster job worker. main.js and earn-cli.js each had
// a copy, and the copies drifted into different bugs. The GUI pinged and joined
// one server while it registered and polled at another, and never retried a
// register that failed. The CLI signed outside its try, so a corrupt key was an
// unhandled rejection from a timer, and it never retried a failed register
// either. One copy now. No Electron here: esbuild bundles the CLI.
//
// Every call is best effort and never throws. An unreachable server must not
// stop the local model or the miner.

const nodeProto = require('../shared/node');
const { NODE } = require('../shared/config');
const { postJson, streamChatCompletion } = require('./io');
const { JobWorker } = require('./jobWorker');
const probe = require('./probe');
const nodeStore = require('./nodeStore');

// The server this node talks to: the one `connect --server` stored in node.json
// (which both shells share), else the default. Every call goes through this.
function serverOf(node) {
  return (node && node.serverUrl) || NODE.serverUrl;
}

// `log(level, line)` is the shell's log ('info' or 'error'). `linkHint` ends the
// line an unlinked node logs once it is registered: each shell links its own way.
function createNodeClient({ log, linkHint }) {
  // Register this box as an unclaimed node so the server will answer its job
  // polls. Signed like a ping; the server derives the node id from the public
  // key and may enrol it under a different one (see nodeProto.adoptedNodeId),
  // which is persisted so later signed calls name a row that exists. Workers
  // already built keep the old id until they are rebuilt; the server reassigns
  // an id only in rare cases. Signing is inside the try: a corrupt key in
  // node.json throws there. Resolves whether it registered.
  async function register(node) {
    try {
      const body = nodeProto.buildPingBody({
        nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey,
        timestamp: Date.now(), telemetry: { name: node.name || undefined },
      });
      const res = await postJson(serverOf(node) + '/api/nodes/register', body, 15000);
      if (res.status !== 200) return false;
      const adopted = nodeProto.adoptedNodeId(node.nodeId, res.data && res.data.nodeId);
      if (adopted !== node.nodeId) {
        node.nodeId = adopted;
        nodeStore.saveNode(Object.assign({}, nodeStore.loadNode() || node, { nodeId: adopted }));
        log('info', 'node id updated by the network to ' + adopted);
      }
      return true;
    } catch (e) {
      return false;
    }
  }

  // An unlinked node needs a row server-side before /jobs/poll will answer it.
  // enrol() registers it, and is called again before every keep-alive ping
  // until that works. Both shells used to try once per process, so a server
  // that was down at that moment left the node polling for jobs it could never
  // be given. A linked node is enrolled already. Resolves whether it is.
  let enrolledId = null;
  let enrolling = null;
  let failureLogged = false;
  function enrol(node) {
    if (node.connected || enrolledId === node.nodeId) return Promise.resolve(true);
    if (!enrolling) {
      enrolling = register(node).then((ok) => {
        enrolling = null;
        if (ok) {
          enrolledId = node.nodeId;
          log('info', 'serving public jobs as an unlinked node (' + node.nodeId + ') — ' + linkHint);
        } else if (!failureLogged) {
          // Once: the retry runs every few minutes.
          failureLogged = true;
          log('error', 'could not register with the network — running the LLM locally only for now, and will keep trying');
        }
        return ok;
      });
    }
    return enrolling;
  }

  // One signed ping. `telemetry` may be sparse: fields left undefined keep the
  // server's stored values. An unlinked node is enrolled first if it is not
  // yet, which is the retry above. Resolves { status }, or { error } when the
  // ping could not be sent or signed.
  async function ping(node, telemetry) {
    await enrol(node);
    try {
      const body = nodeProto.buildPingBody({
        nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey,
        timestamp: Date.now(), telemetry,
      });
      const res = await postJson(serverOf(node) + '/api/nodes/ping', body, 15000);
      return { status: res.status };
    } catch (e) {
      return { error: e.message };
    }
  }

  // The card's name never changes, so it is probed once, not on every ping.
  let device = null;
  function deviceName() {
    if (!device) {
      device = (async () => {
        try {
          const info = await probe.detectGpuInfo();
          return (info && info.name) || null;
        } catch (e) {
          return null;
        }
      })();
    }
    return device;
  }

  // Full telemetry for a node that serves: the model it serves (`model` is a
  // catalogue entry), whether it answers, its rate and running jobs, the card
  // and its VRAM.
  async function telemetry({ model, ready, tokensPerSec, activeJobs, name }) {
    let vram = null;
    try { vram = await probe.detectVram(); } catch (e) { /* unknown this time */ }
    return nodeProto.buildTelemetry({
      model: model.name, quant: model.quant, device: await deviceName(), vram,
      tokensPerSec, ready, activeJobs, name,
    });
  }

  // Link this machine to an account with a pairing token. Resolves one of:
  //   { ok: true, nodeId, user }  linked; nodeId is the id the server enrolled
  //                               it under, which the caller saves
  //   { ok: false, unreachable: true, error }
  //   { ok: false, status, message, raw }  refused; message is the server's
  async function join(node, token, name) {
    const body = nodeProto.buildJoinBody({ token, nodeId: node.nodeId, publicKey: node.publicKey, name });
    let res;
    try {
      res = await postJson(serverOf(node) + '/api/nodes/join', body, 20000);
    } catch (e) {
      return { ok: false, unreachable: true, error: e.message };
    }
    if (res.status !== 200 && res.status !== 201) {
      return { ok: false, status: res.status, message: (res.data && res.data.error) || null, raw: res.raw };
    }
    return {
      ok: true,
      nodeId: nodeProto.adoptedNodeId(node.nodeId, res.data && res.data.nodeId),
      user: (res.data && res.data.user) || null,
    };
  }

  // A cluster job worker for one llama-server, signed as `node`. The caller
  // starts it. `baseUrl` may be a thunk: demand mode builds its worker before
  // any server exists. `gate`, when given, makes the worker WAKE the model
  // rather than assume one is loaded. `servingModel` names the model jobs ran
  // on (see JobWorker).
  function makeJobWorker(node, { baseUrl, gate, servingModel }) {
    const urlOf = typeof baseUrl === 'function' ? baseUrl : () => baseUrl;
    const stream = (chatBody, h) => streamChatCompletion(urlOf(), chatBody, h.onDelta, h.onReasoning).done;
    const w = new JobWorker({
      identity: { nodeId: node.nodeId, publicKey: node.publicKey, secretKey: node.secretKey },
      serverUrl: serverOf(node),
      post: (url, body, opts) => postJson(url, body, 30000, opts),
      runJob: gate
        ? async (chatBody, h) => {
          // begin()/end() bracket the WHOLE job, not just the wake. The gate
          // hands the card back only with nothing in flight, and a cluster job
          // skips the HTTP handler that otherwise counts it.
          gate.begin();
          try {
            await gate.ensureServing();
            return await stream(chatBody, h);
          } finally { gate.end(); }
        }
        : stream,
      servingModel,
    });
    // The 'error' listener is mandatory: a listener-less EventEmitter 'error'
    // throws. Logged in both shells; the GUI used to drop these, so a node
    // whose polls kept failing looked the same as an idle one.
    w.on('error', (e) => log('error', 'job poll failed: ' + e.message + ' (retrying)'));
    w.on('job', ({ id }) => log('info', 'cluster job ' + id + ' — running locally'));
    w.on('done', ({ id }) => log('info', 'cluster job ' + id + ' — done'));
    w.on('failed', ({ id, error }) => log('error', 'cluster job ' + id + ' failed: ' + error));
    return w;
  }

  return { enrol, ping, deviceName, telemetry, join, makeJobWorker };
}

module.exports = { createNodeClient, serverOf };
