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
    this.log = opts.log || (() => {});
    this.gate = opts.gate || new LlmGate(opts);
    this.server = null;
    this.timer = null;
  }

  // Answer a probe from the gate's own state, without waking the model. Shape
  // matches llama-server's so existing dashboards and health checks keep working.
  _passive(req, res) {
    const up = this.gate.isLlmReady();
    const path = String(req.url || '').split('?')[0];
    let body;
    if (path.startsWith('/health')) {
      body = { status: up ? 'ok' : 'loading', gate: this.gate.state };
    } else if (path.startsWith('/v1/models') || path.startsWith('/models')) {
      body = { object: 'list', data: [{ id: this.modelName, object: 'model', owned_by: 'local' }] };
    } else {
      body = { gate: this.gate.state, llm_up: up };
    }
    const buf = Buffer.from(JSON.stringify(body));
    res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': buf.length });
    res.end(buf);
  }

  _forward(req, res, body) {
    const opts = {
      host: this.upstreamHost, port: this.upstreamPort,
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
      this._forward(req, res, body);
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
