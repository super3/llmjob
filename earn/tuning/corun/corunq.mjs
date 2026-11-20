// Co-run and switch-time harness for issue #312, for the Pearl core or the Quantus core.
//
//   node corunq.mjs <mode> <secs> <warm> <tag>
//
// Modes:
//   miner      the miner alone (MINER=pearl|quantus)
//   llm        decode alone: short prompt, NTOK tokens a request, back to back
//   both       miner and decode in this one process (one CUDA context)
//   pause      switch times the way the app pauses today: LLM loaded; per request stop the miner, answer, then
//              start the miner again and time how long it takes to get back to full rate. REPS cycles, after REPS
//              requests with no miner at all as the baseline.
//   coldllm    from process start: load the model and give the first token
//   coldminer  from process start: open the miner and reach full rate
//
// Env: MINER (pearl|quantus), PCORE (pearl_core.node), QCORE (quantus_cuda.node), MODEL, NTOK (512),
//      SOLO (the miner's solo rate, for "full rate" = 95% of it), PROFILE_JSON, READY_FILE (two-process runs),
//      PRIO / ISOLATE (llama.cpp stream hooks from the Pearl core), PEARL_FREE_SMS, QUANTUS_CUDA_GRID.
// Writes one JSON line to results.jsonl beside this file. Rates: Pearl TH/s, Quantus GH/s.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';
import { fileURLToPath } from 'url';
const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const [mode, secsS, warmS, tag] = process.argv.slice(2);
const secs = Number(secsS || 90), warm = Number(warmS || 30);
const KIND = process.env.MINER || 'pearl';
const PCORE = process.env.PCORE || path.join(HERE, 'pearl_core.node');
const QCORE = process.env.QCORE || path.join(HERE, 'quantus_cuda.node');
const MODEL = process.env.MODEL || path.join(process.env.APPDATA || '', 'llmjob-earn', 'llm', 'gemma-4-E4B-it-Q4_K_M.gguf');
const NTOK = Number(process.env.NTOK || 512);
const SOLO = Number(process.env.SOLO || 0);
const REPS = Number(process.env.REPS || 4);
process.env.CUDA_DEVICE_ORDER = 'PCI_BUS_ID';
const sinceStart = () => performance.now();   // ms since the process started
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const out = { tag, mode, miner: KIND, secs, warm };
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const r1 = (x) => (x == null ? null : +x.toFixed(1));

// ---------------------------------------------------------------- miners
// start() opens the core and starts the search; stop() stops it and frees the card. `samples` gets
// { t, v } rate readings: Pearl's own hashrate ticks (about two a second), or Quantus scans summed
// into half-second buckets. `onSample` sees each one.
function pearlMiner() {
  const PROFILE = JSON.parse(fs.readFileSync(process.env.PROFILE_JSON || path.join(HERE, 'profile.json'), 'utf8'));
  const addon = require(PCORE);
  const m = { samples: [], onSample: null, core: null };
  m.start = () => {
    const c = addon.createCore(PROFILE, { cardK: true, checkHits: true });
    c.on('hashrate', (th) => { const s = { t: sinceStart(), v: th }; m.samples.push(s); if (m.onSample) m.onSample(s); });
    c.on('error', (e) => { console.error('core error', e); process.exit(2); });
    c.on('hit', () => {});
    c.setJob({ header: Buffer.alloc(76, 0xA5), target: 0n, jobId: 'bench-0' });
    m.core = c; m.device = c.device;
  };
  m.stop = async () => { const c = m.core; m.core = null; if (c) c.stop(); };
  return m;
}

function quantusMiner() {
  const addon = require(QCORE);
  const midstate = Buffer.alloc(96);
  for (let i = 0; i < 12; i++) midstate.writeBigUInt64LE(BigInt(i * 7919 + 12345), i * 8);
  const tail = Buffer.alloc(24, 0x5A);
  const zero = Buffer.alloc(8);
  const m = { samples: [], onSample: null, core: null, counted: [] };
  let running = false, loop = null, counter = 0n;
  m.start = () => {
    const c = addon.createCore({ device: 0 });
    m.core = c; m.device = c.device; running = true;
    loop = (async () => {
      let batch = 1 << 22, bStart = sinceStart(), bCount = 0;
      while (running) {
        const a = sinceStart();
        const s = new Uint8Array(8); const sb = Buffer.from(s.buffer); sb.writeBigUInt64BE(counter);
        await c.scan({ midstate, tail, start: sb, count: batch, targetHigh: zero, maxHits: 16 });
        const b = sinceStart();
        counter += BigInt(batch);
        m.counted.push({ t: b, n: batch });
        bCount += batch;
        if (b - bStart >= 500) {
          const smp = { t: b, v: bCount / ((b - bStart) / 1000) / 1e9 };
          m.samples.push(smp); if (m.onSample) m.onSample(smp);
          bStart = b; bCount = 0;
        }
        // As the engine does: size the next scan to about 100 ms.
        const next = Math.round(batch * 100 / Math.max(b - a, 1));
        batch = Math.max(1, Math.min(2 ** 32 - 1, batch * 16, next));
      }
    })();
  };
  m.stop = async () => {
    running = false;
    if (loop) await loop;
    loop = null;
    const c = m.core; m.core = null; if (c) c.close();
  };
  return m;
}
const makeMiner = () => (KIND === 'quantus' ? quantusMiner() : pearlMiner());

// The miner's mean rate over [a, b] ms since start.
function rateIn(m, a, b) {
  if (KIND === 'quantus') {
    const n = m.counted.filter((x) => x.t > a && x.t <= b).reduce((s, x) => s + x.n, 0);
    return n / ((b - a) / 1000) / 1e9;
  }
  return mean(m.samples.filter((x) => x.t > a && x.t <= b).map((x) => x.v));
}

// Resolves with ms from `from` to the first reading at 95% of SOLO or more (null after `limitMs`).
function untilFull(m, from, limitMs = 60000) {
  return new Promise((resolve) => {
    const done = (v) => { clearTimeout(t); m.onSample = null; resolve(v); };
    const t = setTimeout(() => done(null), limitMs);
    m.onSample = (s) => { if (SOLO && s.v >= 0.95 * SOLO) done(s.t - from); };
  });
}

// ---------------------------------------------------------------- the LLM
let llm = null;
async function loadLlm() {
  const t = { a: sinceStart() };
  const { getLlama } = await import('node-llama-cpp');
  const llama = await getLlama({ gpu: 'cuda', build: 'never' });
  t.lib = sinceStart();
  if (process.env.ISOLATE) out.isolate = require(PCORE).isolateDefaultStreams('ggml-cuda.dll', 'cudart64_12.dll', !!process.env.PRIO);
  else if (process.env.PRIO) out.prio = require(PCORE).raiseStreamPriority('ggml-cuda.dll', 'cudart64_12.dll');
  const model = await llama.loadModel({ modelPath: MODEL, gpuLayers: 'max' });
  const context = await model.createContext({ contextSize: 2048, batchSize: 512 });
  t.loaded = sinceStart();
  llm = { llama, model, context, seq: context.getSequence() };
  llm.prompt = model.tokenize('<start_of_turn>user\nWrite a long story about a lighthouse keeper.<end_of_turn>\n<start_of_turn>model\n', true);
  out.llmGpu = llama.gpu;
  return t;
}
// One request: { first (ms to the first token from `from`), n, end }.
async function request(ntok, from = sinceStart(), stopAt = Infinity, onTok = null) {
  const { seq, prompt } = llm;
  let first = null, n = 0;
  await seq.clearHistory();
  for await (const tok of seq.evaluate(prompt, { temperature: 0.8, seed: 1 })) {
    n++;
    if (n === 1) first = sinceStart();
    if (onTok) onTok();
    if (n >= ntok || sinceStart() > stopAt) break;
  }
  return { first, ttft: first == null ? null : first - from, n, end: sinceStart() };
}

const stats = (xs) => {
  const v = xs.filter((x) => x != null).sort((a, b) => a - b);
  if (!v.length) return null;
  const med = v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
  return { n: v.length, medS: +(med / 1000).toFixed(2), minS: +(v[0] / 1000).toFixed(2), maxS: +(v[v.length - 1] / 1000).toFixed(2) };
};

function finish() {
  console.log(JSON.stringify(out));
  fs.appendFileSync(path.join(HERE, 'results.jsonl'), JSON.stringify(out) + '\n');
  process.exit(0);
}

// ---------------------------------------------------------------- cold starts
if (mode === 'coldllm') {
  const t = await loadLlm();
  const r = await request(4, 0);
  out.libMs = Math.round(t.lib); out.loadedMs = Math.round(t.loaded); out.firstTokenMs = Math.round(r.first);
  finish();
}
if (mode === 'coldminer') {
  const m = makeMiner();
  const tAddon = sinceStart();
  const full = untilFull(m, 0, secs * 1000);
  m.start();
  const tOpen = sinceStart();
  const ms = await full;
  out.addonMs = Math.round(tAddon); out.openMs = Math.round(tOpen); out.fullMs = ms == null ? null : Math.round(ms);
  out.firstSampleMs = m.samples.length ? Math.round(m.samples[0].t) : null;
  await m.stop();
  finish();
}

// ---------------------------------------------------------------- pause-mode switch times
if (mode === 'pause') {
  await loadLlm();
  await request(16);   // warm the model
  const alone = [];
  for (let i = 0; i < REPS; i++) { const r = await request(32); alone.push(r.ttft); await sleep(1000); }
  const m = makeMiner();
  const full0 = untilFull(m, sinceStart());
  const tm = sinceStart();
  m.start();
  out.warmProcessOpenFullMs = await full0;
  void tm;
  await sleep(warm * 1000);
  const cyc = [];
  for (let i = 0; i < REPS; i++) {
    const t0 = sinceStart();
    await m.stop();
    const stopMs = sinceStart() - t0;
    const r = await request(32, t0);
    const tR = sinceStart();
    const full = untilFull(m, tR);
    m.start();
    const fullMs = await full;
    cyc.push({ stopMs: Math.round(stopMs), ttftMs: Math.round(r.ttft), resumeFullMs: fullMs == null ? null : Math.round(fullMs) });
    await sleep(secs * 1000);
  }
  await m.stop();
  out.aloneTtftMs = alone.map(Math.round);
  out.cycles = cyc;
  out.aloneTtft = stats(alone);
  out.pauseTtft = stats(cyc.map((c) => c.ttftMs));
  out.stop = stats(cyc.map((c) => c.stopMs));
  out.resumeFull = stats(cyc.map((c) => c.resumeFullMs));
  finish();
}

// ---------------------------------------------------------------- co-run measurements
if (mode === 'llm' || mode === 'both') await loadLlm();
// Two-process runs: the llm process writes READY_FILE once loaded; the miner process waits for it.
if (process.env.READY_FILE) {
  if (llm) fs.writeFileSync(process.env.READY_FILE, String(Date.now()));
  else while (!fs.existsSync(process.env.READY_FILE)) await sleep(200);
}
const tStart = sinceStart();
const wA = tStart + warm * 1000, wB = tStart + (warm + secs) * 1000;
let miner = null;
if (mode === 'miner' || mode === 'both') { miner = makeMiner(); miner.start(); }
let winToks = 0; const runs = []; const ttfts = [];
if (llm) {
  while (sinceStart() < wB) {
    const a = sinceStart();
    const r = await request(NTOK, a, wB, () => { const t = sinceStart(); if (t > wA && t < wB) winToks++; });
    if (a > wA && r.ttft != null && r.first < wB) ttfts.push(r.ttft);
    if (a > wA && r.end < wB && r.n > 1) runs.push({ tps: (r.n - 1) / ((r.end - r.first) / 1000), n: r.n, wall: (r.end - a) / 1000 });
  }
} else await sleep(wB - sinceStart() + 50);
if (miner) {
  out.rate = +rateIn(miner, wA, wB).toFixed(4);
  out.unit = KIND === 'quantus' ? 'GH/s' : 'TH/s';
  out.samples = miner.samples.length; out.device = miner.device;
  await miner.stop();
}
if (llm) {
  out.windowTokTps = r1(winToks / secs);
  out.answers = runs.length;
  out.decodeTps = runs.length ? r1(mean(runs.map((r) => r.tps))) : null;
  out.ttftMs = ttfts.map(Math.round);
  out.ttft = stats(ttfts);
}
if (process.env.PRIO) out.hookedStreams = require(PCORE).streamHookCount();
const iso = (ms) => new Date(performance.timeOrigin + ms).toISOString();
out.windowStartUtc = iso(wA); out.windowEndUtc = iso(wB);
finish();
