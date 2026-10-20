'use strict';

// The node's side of the network, shared by both shells (src/main/nodeClient.js).
// The network, the probes and node.json are mocked; the signing and the job
// worker are real.

jest.mock('../src/main/io', () => ({ postJson: jest.fn(), streamChatCompletion: jest.fn() }));
jest.mock('../src/main/probe', () => ({ detectGpuInfo: jest.fn(), detectVram: jest.fn() }));
jest.mock('../src/main/nodeStore', () => ({ loadNode: jest.fn(), saveNode: jest.fn() }));

const nacl = require('tweetnacl');
const naclUtil = require('tweetnacl-util');
const io = require('../src/main/io');
const probe = require('../src/main/probe');
const nodeStore = require('../src/main/nodeStore');
const nodeProto = require('../src/shared/node');
const { NODE, LLM } = require('../src/shared/config');
const { JobWorker } = require('../src/main/jobWorker');
const { createNodeClient, serverOf } = require('../src/main/nodeClient');

const KEYS = nodeProto.generateKeypair();
const node = (extra) => Object.assign({
  nodeId: 'abc123', publicKey: KEYS.publicKey, secretKey: KEYS.secretKey, name: 'rig', connected: false,
}, extra);
const flush = () => new Promise((r) => setImmediate(r));

let logs;
function client() {
  logs = [];
  return createNodeClient({ log: (level, line) => logs.push([level, line]), linkHint: 'link it' });
}
const posted = (suffix) => io.postJson.mock.calls.filter((c) => c[0].endsWith(suffix));
const signedBy = (body) => nacl.sign.detached.verify(
  naclUtil.decodeUTF8(body.nodeId + ':' + body.timestamp),
  naclUtil.decodeBase64(body.signature), naclUtil.decodeBase64(KEYS.publicKey));

beforeEach(() => {
  jest.clearAllMocks();
  io.postJson.mockResolvedValue({ status: 200, data: {} });
});

describe('serverOf', () => {
  // `connect --server` stores the server in the node.json both shells share.
  it('is the stored server, else the default', () => {
    expect(serverOf(node({ serverUrl: 'https://s.example' }))).toBe('https://s.example');
    expect(serverOf(node())).toBe(NODE.serverUrl);
    expect(serverOf(null)).toBe(NODE.serverUrl);
  });
});

describe('enrol', () => {
  it('is a no-op for a linked node', async () => {
    await expect(client().enrol(node({ connected: true }))).resolves.toBe(true);
    expect(io.postJson).not.toHaveBeenCalled();
  });

  it('registers an unlinked node once, signed, at its server', async () => {
    const c = client();
    const n = node({ serverUrl: 'https://s.example', name: null });
    // Two callers at once share one request.
    const [a, b] = await Promise.all([c.enrol(n), c.enrol(n)]);
    expect([a, b]).toEqual([true, true]);
    expect(io.postJson).toHaveBeenCalledTimes(1);
    const [url, body, timeout] = io.postJson.mock.calls[0];
    expect(url).toBe('https://s.example/api/nodes/register');
    expect(timeout).toBe(15000);
    expect(body).toMatchObject({ nodeId: 'abc123', publicKey: KEYS.publicKey });
    expect(body.name).toBeUndefined();
    expect(signedBy(body)).toBe(true);
    expect(logs).toEqual([['info', 'serving public jobs as an unlinked node (abc123) — link it']]);

    await expect(c.enrol(n)).resolves.toBe(true);   // enrolled: nothing more
    expect(io.postJson).toHaveBeenCalledTimes(1);
  });

  // The server decides which id a machine is enrolled under (see
  // nodeProto.adoptedNodeId), and the answer is persisted.
  it('adopts and saves an id the server hands back', async () => {
    io.postJson.mockResolvedValue({ status: 200, data: { nodeId: 'a1b2c3d4e5f60789' } });
    nodeStore.loadNode.mockReturnValue({ nodeId: 'abc123', user: 'kept' });
    const n = node();
    await expect(client().enrol(n)).resolves.toBe(true);
    expect(n.nodeId).toBe('a1b2c3d4e5f60789');
    expect(nodeStore.saveNode).toHaveBeenCalledWith({ nodeId: 'a1b2c3d4e5f60789', user: 'kept' });
    expect(logs[0]).toEqual(['info', 'node id updated by the network to a1b2c3d4e5f60789']);

    // With nothing on disk the node itself is saved.
    nodeStore.loadNode.mockReturnValue(null);
    const m = node();
    await client().enrol(m);
    expect(nodeStore.saveNode).toHaveBeenLastCalledWith(expect.objectContaining({
      nodeId: 'a1b2c3d4e5f60789', publicKey: KEYS.publicKey,
    }));
  });

  // Both shells used to try once per process.
  it('tries again after a failure, and says so once', async () => {
    const c = client();
    const n = node();
    io.postJson.mockResolvedValueOnce({ status: 400, data: { error: 'Node key mismatch' } });
    await expect(c.enrol(n)).resolves.toBe(false);
    io.postJson.mockRejectedValueOnce(new Error('offline'));
    await expect(c.enrol(n)).resolves.toBe(false);
    expect(logs).toEqual([['error', expect.stringContaining('could not register with the network')]]);
    await expect(c.enrol(n)).resolves.toBe(true);
    expect(logs.pop()).toEqual(['info', expect.stringContaining('serving public jobs as an unlinked node')]);
    expect(nodeStore.saveNode).not.toHaveBeenCalled();
  });

  // Signing is inside the try: a corrupt key fails the register, quietly.
  it('fails, without throwing, on a corrupt key', async () => {
    await expect(client().enrol(node({ secretKey: 'not-a-key' }))).resolves.toBe(false);
    expect(io.postJson).not.toHaveBeenCalled();
  });
});

describe('ping', () => {
  it('posts a signed ping with the telemetry to the node\'s server', async () => {
    const res = await client().ping(node({ connected: true, serverUrl: 'https://s.example' }), { name: 'rig', tps: 3 });
    expect(res).toEqual({ status: 200 });
    const [url, body, timeout] = io.postJson.mock.calls[0];
    expect(url).toBe('https://s.example/api/nodes/ping');
    expect(timeout).toBe(15000);
    expect(body).toMatchObject({ nodeId: 'abc123', name: 'rig', tps: 3 });
    expect(signedBy(body)).toBe(true);
  });

  it('enrols an unlinked node before pinging', async () => {
    await client().ping(node(), {});
    expect(io.postJson.mock.calls.map((c) => c[0])).toEqual([
      NODE.serverUrl + '/api/nodes/register', NODE.serverUrl + '/api/nodes/ping',
    ]);
  });

  it('joins a register still in flight instead of sending another', async () => {
    let answer;
    io.postJson.mockImplementationOnce(() => new Promise((r) => { answer = r; }));
    const c = client();
    const n = node();
    const enrolled = c.enrol(n);
    const pinged = c.ping(n, {});
    await flush();
    expect(posted('/api/nodes/register')).toHaveLength(1);
    expect(posted('/api/nodes/ping')).toHaveLength(0);   // waits for it
    answer({ status: 200, data: {} });
    await Promise.all([enrolled, pinged]);
    expect(posted('/api/nodes/register')).toHaveLength(1);
    expect(posted('/api/nodes/ping')).toHaveLength(1);
  });

  // Pings run from timers in both shells: nothing may reject there.
  it('resolves an error instead of rejecting', async () => {
    io.postJson.mockRejectedValueOnce(new Error('net down'));
    await expect(client().ping(node({ connected: true }), {})).resolves.toEqual({ error: 'net down' });
    await expect(client().ping(node({ connected: true, secretKey: 'not-a-key' }), {}))
      .resolves.toEqual({ error: expect.any(String) });
  });
});

describe('deviceName and telemetry', () => {
  it('probes the card once', async () => {
    probe.detectGpuInfo.mockResolvedValue({ name: 'RTX 5090', count: 1 });
    const c = client();
    expect(await c.deviceName()).toBe('RTX 5090');
    expect(await c.deviceName()).toBe('RTX 5090');
    expect(probe.detectGpuInfo).toHaveBeenCalledTimes(1);
  });

  it('is null when the probe finds nothing or fails', async () => {
    probe.detectGpuInfo.mockResolvedValue(null);
    expect(await client().deviceName()).toBeNull();
    probe.detectGpuInfo.mockRejectedValue(new Error('nvidia-smi exploded'));
    expect(await client().deviceName()).toBeNull();
    probe.detectGpuInfo.mockImplementation(() => { throw new Error('spawn fail'); });
    expect(await client().deviceName()).toBeNull();
  });

  it('builds full telemetry from the shell\'s state, the card and its VRAM', async () => {
    probe.detectGpuInfo.mockResolvedValue({ name: 'RTX 5090' });
    probe.detectVram.mockResolvedValue({ totalMb: 32000, usedMb: 1000 });
    const t = await client().telemetry({ model: LLM.model, ready: true, tokensPerSec: 41.5, activeJobs: 2, name: 'rig' });
    expect(t).toEqual(nodeProto.buildTelemetry({
      model: LLM.model.name, quant: LLM.model.quant, device: 'RTX 5090',
      vram: { totalMb: 32000, usedMb: 1000 }, tokensPerSec: 41.5, ready: true, activeJobs: 2, name: 'rig',
    }));
  });

  it('sends no VRAM when it cannot be read', async () => {
    probe.detectGpuInfo.mockResolvedValue(null);
    probe.detectVram.mockRejectedValue(new Error('smi gone'));
    const t = await client().telemetry({ model: LLM.model, ready: false, tokensPerSec: 0, activeJobs: 0 });
    expect(t).toMatchObject({ vramTotal: null, vramUsed: null, device: null, capabilities: [] });
  });
});

describe('join', () => {
  it('links at the node\'s server and returns the enrolled id and user', async () => {
    io.postJson.mockResolvedValue({ status: 201, data: { user: 'alice', nodeId: 'a1b2c3d4e5f60789' } });
    const res = await client().join(node({ serverUrl: 'https://s.example' }), 'tok', 'MyRig');
    expect(res).toEqual({ ok: true, nodeId: 'a1b2c3d4e5f60789', user: 'alice' });
    expect(io.postJson).toHaveBeenCalledWith('https://s.example/api/nodes/join',
      { token: 'tok', nodeId: 'abc123', publicKey: KEYS.publicKey, name: 'MyRig' }, 20000);
  });

  it('keeps the id and has no user when the server says neither', async () => {
    io.postJson.mockResolvedValue({ status: 200, data: null });
    await expect(client().join(node(), 'tok', null)).resolves.toEqual({ ok: true, nodeId: 'abc123', user: null });
  });

  it('reports a refusal with the server\'s reason, if it gave one', async () => {
    io.postJson.mockResolvedValueOnce({ status: 403, data: { error: 'bad token' }, raw: 'x' });
    await expect(client().join(node(), 'tok')).resolves.toEqual({ ok: false, status: 403, message: 'bad token', raw: 'x' });
    io.postJson.mockResolvedValueOnce({ status: 500, data: null, raw: 'boom' });
    await expect(client().join(node(), 'tok')).resolves.toEqual({ ok: false, status: 500, message: null, raw: 'boom' });
  });

  it('reports an unreachable server', async () => {
    io.postJson.mockRejectedValue(new Error('ECONNREFUSED'));
    await expect(client().join(node(), 'tok')).resolves.toEqual({ ok: false, unreachable: true, error: 'ECONNREFUSED' });
  });
});

describe('makeJobWorker', () => {
  it('signs as the node, polls its server and streams jobs to the local model', async () => {
    const serving = () => LLM.model;
    const w = client().makeJobWorker(node({ serverUrl: 'https://s.example' }), { baseUrl: 'http://127.0.0.1:8081', servingModel: serving });
    expect(w).toBeInstanceOf(JobWorker);
    expect(w.identity).toEqual({ nodeId: 'abc123', publicKey: KEYS.publicKey, secretKey: KEYS.secretKey });
    expect(w.serverUrl).toBe('https://s.example');
    expect(w.servingModel).toBe(serving);

    // The abort signal reaches the request (see JobWorker.stop).
    const signal = new AbortController().signal;
    await w.post('https://x/api', { a: 1 }, { signal });
    expect(io.postJson).toHaveBeenCalledWith('https://x/api', { a: 1 }, 30000, { signal });

    io.streamChatCompletion.mockReturnValue({ done: Promise.resolve({ finishReason: 'stop' }) });
    const h = { onDelta: jest.fn(), onReasoning: jest.fn() };
    await expect(w.runJob({ messages: [] }, h)).resolves.toEqual({ finishReason: 'stop' });
    expect(io.streamChatCompletion).toHaveBeenCalledWith('http://127.0.0.1:8081', { messages: [] }, h.onDelta, h.onReasoning);
  });

  it('logs what the worker does, poll failures included', () => {
    const w = client().makeJobWorker(node(), { baseUrl: 'http://h' });
    w.emit('error', new Error('the server refused the poll (HTTP 404)'));
    w.emit('job', { id: 'j1' });
    w.emit('done', { id: 'j1' });
    w.emit('failed', { id: 'j2', error: 'oom' });
    expect(logs).toEqual([
      ['error', 'job poll failed: the server refused the poll (HTTP 404) (retrying)'],
      ['info', 'cluster job j1 — running locally'],
      ['info', 'cluster job j1 — done'],
      ['error', 'cluster job j2 failed: oom'],
    ]);
  });

  // Demand mode: no server exists when the worker is built, so the URL is read
  // per job, and the job wakes the model and holds the card for its whole run.
  it('with a gate, wakes the model and brackets the whole job', async () => {
    let url = null;
    const gate = { begin: jest.fn(), end: jest.fn(), ensureServing: jest.fn(async () => { url = 'http://127.0.0.1:8090'; }) };
    const w = client().makeJobWorker(node(), { baseUrl: () => url, gate });
    io.streamChatCompletion.mockReturnValue({ done: Promise.resolve() });
    await w.runJob({ messages: [] }, { onDelta: () => {} });
    expect(gate.begin).toHaveBeenCalled();
    expect(io.streamChatCompletion.mock.calls[0][0]).toBe('http://127.0.0.1:8090');
    expect(gate.end).toHaveBeenCalledTimes(1);

    io.streamChatCompletion.mockReturnValue({ done: Promise.reject(new Error('model died')) });
    await expect(w.runJob({ messages: [] }, { onDelta: () => {} })).rejects.toThrow('model died');
    expect(gate.end).toHaveBeenCalledTimes(2);   // or the card never mines again
  });
});
