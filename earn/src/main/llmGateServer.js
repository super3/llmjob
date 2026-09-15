'use strict';

// The public endpoint for demand-driven auto mode.
//
// It owns the port callers use (8000 by default) and puts llama-server behind it,
// so the model can be stopped and started underneath without the endpoint ever
// going away. A caller sees one stable address whether the card is currently
// mining or serving; the only visible difference is that the first request after
// a quiet period takes ~4 s while the model loads.
//
// The request is HELD during that load rather than refused. Returning a connection
// error and asking callers to retry would push the switch into every client.

const http = require('http');
const { LlmGate, classifyPath } = require('../shared/llmGate');

// How long an idle pooled connection is held open. Has to outlast the pauses a
// caller takes between requests -- an agent thinking or running a tool -- because
// whoever closes first decides whose problem the reset is, and a client cannot
// see it coming.
const KEEPALIVE_MS = 10 * 60 * 1000;

// Hop-by-hop headers: meaningful to one connection, wrong to forward.
const HOP = new Set(['connection', 'keep-alive', 'transfer-encoding', 'upgrade',
  'proxy-authorization', 'proxy-authenticate', 'te', 'trailer']);

function pickHeaders(headers) {
  const out = {};
  for (const [k, v] of Object.entries(headers || {})) {
    if (!HOP.has(k.toLowerCase()) && k.toLowerCase() !== 'host') out[k] = v;
  }
  return out;
}

class LlmGateServer {
  constructor(opts = {}) {
    this.port = opts.port == null ? 8000 : opts.port;
    this.host = opts.host || '0.0.0.0';
    this.upstreamHost = opts.upstreamHost || '127.0.0.1';
    this.upstreamPort = opts.upstreamPort || 8080;
    this.modelName = opts.modelName || 'local';
    // The context window this node serves. Reported on the passive endpoints so a
    // caller detects the same number whether or not the model happens to be
    // loaded -- see _passive.
    this.ctxSize = opts.ctxSize || null;
    // Optional: told when the port could not be bound, so a caller can decide
    // whether that is fatal. Absent means "log it and carry on serving locally".
    this.onListenError = opts.onListenError || null;
    this.log = opts.log || (() => {});
    this.gate = opts.gate || new LlmGate(opts);
    this.server = null;
    this.timer = null;
  }

  // Answer a probe from the gate's own state, without waking the model. Shape
  // matches llama-server's so existing dashboards and health checks keep working.
  //
  // n_ctx is included because callers DETECT the context window from these two
  // endpoints, and they are passive precisely so a probe cannot wake the card.
  // Answering without it meant the reported window depended on whether the model
  // happened to be loaded: a client that asked while mining got no n_ctx, fell
  // back to its own default (commonly 131072), and then compacted at half the
  // window this node actually serves. Same node, same config, half the context,
  // decided by timing.
  _passive(req, res) {
    const up = this.gate.isLlmReady();
    const path = String(req.url || '').split('?')[0];
    const ctx = this.ctxSize;
    let body;
    if (path.startsWith('/health')) {
      body = { status: up ? 'ok' : 'loading', gate: this.gate.state };
    } else if (path.startsWith('/v1/models') || path.startsWith('/models')) {
      const model = { id: this.modelName, object: 'model', owned_by: 'local' };
      // Nested under `meta` to match llama-server's own /v1/models entry, so a
      // client reads the field from the same place either way.
      if (ctx) model.meta = { n_ctx: ctx, n_ctx_train: ctx };
      body = { object: 'list', data: [model] };
    } else if (path.startsWith('/props')) {
      // llama-server reports it under default_generation_settings; mirror that.
      body = { gate: this.gate.state, llm_up: up };
      if (ctx) body.default_generation_settings = { n_ctx: ctx };
    } else {
      body = { gate: this.gate.state, llm_up: up };
    }
    const buf = Buffer.from(JSON.stringify(body));
    res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': buf.length });
    res.end(buf);
  }

  // The upstream port may be a function: the fleet chooses its port at spawn
  // time, so a value read once at construction can be stale by the first request.
  _port() {
    return typeof this.upstreamPort === 'function' ? this.upstreamPort() : this.upstreamPort;
  }

  _forward(req, res, body) {
    const opts = {
      host: this.upstreamHost, port: this._port(),
      method: req.method, path: req.url, headers: pickHeaders(req.headers),
    };
    if (body) opts.headers['content-length'] = Buffer.byteLength(body);
    const up = http.request(opts, (r) => {
      // No `|| 502` fallback: a client response from node's http always carries a
      // statusCode, so the guard would be unreachable rather than defensive.
      res.writeHead(r.statusCode, pickHeaders(r.headers));
      // Piped, not buffered: a streamed completion has to reach the caller token
      // by token, not in one lump when the generation finishes.
      r.pipe(res);
    });
    up.on('error', (e) => {
      if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'upstream: ' + e.message } }));
    });
    if (body) up.end(body); else req.pipe(up);
  }

  // Never rejects: every failure is turned into a response here, so the caller
  // does not have to hold a second error path.
  async _handle(req, res) {
    const kind = classifyPath(req.url);
    if (kind === 'passive') {
      // A probe must never count as activity, whether the model is up or down.
      // When it is down the gate answers from its own state; when it is UP the
      // probe is forwarded so the caller gets real data -- but WITHOUT begin(),
      // because refreshing the quiet clock on every poll pins the card in SERVING
      // for as long as anything is monitoring it. A dashboard polling /health
      // once a second would mean the GPU never goes back to mining.
      if (!this.gate.isLlmReady()) return this._passive(req, res);
      return this._forward(req, res, null);
    }

    this.gate.begin();
    try {
      // Read the body before switching: the client is done sending, and the
      // upstream does not exist yet.
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = chunks.length ? Buffer.concat(chunks) : null;
      await this.gate.ensureServing();
      // The caller may be long gone: a wake is ~4s warm and disk-bound cold, and
      // any client whose timeout is shorter than that has already hung up. 'close'
      // fired while we were awaiting, and node does not replay it for a listener
      // attached afterwards -- so the promise below never settled, the finally
      // never ran, and inFlight stayed at 1 for the life of the process. With
      // shouldRelease() requiring inFlight === 0, that pinned the card in SERVING
      // with the model resident and the miner stopped, permanently.
      //
      // Checked BEFORE forwarding as well as before waiting: there is no point
      // opening an upstream request to pipe into a destroyed socket, which also
      // held an upstream slot until the generation finished.
      if (res.writableEnded || res.destroyed) return;
      this._forward(req, res, body);
      // Safe to attach now: _forward does not await, so nothing can have closed
      // the socket between the guard above and here.
      await new Promise((resolve) => { res.on('close', resolve); res.on('finish', resolve); });
    } catch (e) {
      if (!res.headersSent) res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'llm unavailable: ' + e.message } }));
    } finally {
      this.gate.end();
    }
  }

  start() {
    this.server = http.createServer((req, res) => { this._handle(req, res); });
    // Node's HTTP server defaults are tuned for a short-lived JSON API, and both
    // of the ones below break callers that a direct llama-server endpoint served
    // fine. Neither failure logs anything here, because both are the SERVER
    // hanging up on a healthy client:
    //
    //   keepAliveTimeout, default 5s. An agent pools connections and pauses
    //   between turns to think or run a tool. The gate closed the idle socket
    //   underneath it, so the next request raced onto a half-closed connection
    //   and the client saw ECONNRESET / 'socket hang up'. A proxy's keep-alive
    //   has to outlast its clients', not undercut it.
    //
    //   requestTimeout, default 300s. A 50-80k token prompt re-prefills at
    //   ~730 tok/s and then generates at ~31 tok/s; observed totals on this rig
    //   reach 454s. There is no sane ceiling to put on a generation, so there
    //   is none.
    //
    // headersTimeout must exceed keepAliveTimeout or node warns and the larger
    // value never takes effect.
    this.server.keepAliveTimeout = KEEPALIVE_MS;
    this.server.headersTimeout = KEEPALIVE_MS + 60000;
    this.server.requestTimeout = 0;
    // A listen failure is an 'error' event, not a throw. Unhandled, it is an
    // uncaught exception -- and this process is also the miner, so anything
    // already on the port took mining down with it. Nothing bound this port
    // before, so every node that gains a gate meets this for the first time.
    this.server.on('error', (e) => {
      this.log('gate could not bind :' + this.port + ' — ' + e.message
        + ' (continuing without the public endpoint)');
      this.server = null;
      if (this.onListenError) this.onListenError(e);
    });
    this.server.listen(this.port, this.host);
    this.timer = setInterval(() => {
      if (this.gate.shouldRelease()) {
        this.log('no requests for ' + Math.round(this.gate.quietFor() / 1000)
          + 's — handing the GPU back to mining');
        this.gate.ensureMining().catch(() => {});
      }
    }, 2000);
    if (this.timer.unref) this.timer.unref();
    return this;
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
    if (this.server) this.server.close();
    this.timer = null; this.server = null;
  }
}

module.exports = { LlmGateServer, pickHeaders };
