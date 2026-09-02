'use strict';

const {
  LlmGate, classifyPath, MINING, SERVING,
} = require('../src/shared/llmGate');

describe('classifyPath', () => {
  test('inference endpoints are demand', () => {
    for (const p of ['/completion', '/v1/chat/completions', '/v1/embeddings',
      '/v1/chat/completions?stream=true', '/infill', '/rerank']) {
      expect(classifyPath(p)).toBe('wake');
    }
  });

  // The whole design rests on this. Monitoring polls /health continuously -- our
  // dashboard does it every second -- so if a probe counted as demand the model
  // would wake on the first poll and never hand the card back, and auto mode
  // would silently become llm mode.
  test('probes are NOT demand', () => {
    for (const p of ['/health', '/v1/models', '/models', '/props', '/metrics', '/slots']) {
      expect(classifyPath(p)).toBe('passive');
    }
  });

  test('an unknown path is treated as demand rather than 404d while the model is down', () => {
    expect(classifyPath('/v1/some-new-endpoint')).toBe('wake');
  });

  test('a missing path does not throw', () => {
    expect(classifyPath()).toBe('wake');
    expect(classifyPath(null)).toBe('wake');
  });
});

function mkGate(over = {}) {
  const calls = [];
  let ready = false;
  const g = new LlmGate(Object.assign({
    quietMs: 1000,
    isLlmReady: () => ready,
    startLlm: async () => { calls.push('startLlm'); ready = true; },
    stopLlm: async () => { calls.push('stopLlm'); ready = false; },
    startMiner: async () => { calls.push('startMiner'); },
    stopMiner: async () => { calls.push('stopMiner'); },
  }, over));
  return { g, calls, setReady: (v) => { ready = v; } };
}

describe('LlmGate', () => {
  test('starts out mining', () => {
    expect(mkGate().g.state).toBe(MINING);
  });

  test('waking stops the miner BEFORE starting the LLM', async () => {
    // Order matters: the miner holds VRAM the model needs, and a suspended
    // process would keep it, so the miner must actually exit first.
    const { g, calls } = mkGate();
    await g.ensureServing();
    expect(calls).toEqual(['stopMiner', 'startLlm']);
    expect(g.state).toBe(SERVING);
  });

  test('concurrent requests share one transition instead of starting two servers', async () => {
    const { g, calls } = mkGate();
    await Promise.all([g.ensureServing(), g.ensureServing(), g.ensureServing()]);
    expect(calls.filter((c) => c === 'startLlm')).toHaveLength(1);
  });

  test('releasing stops the LLM then restarts the miner', async () => {
    const { g, calls } = mkGate();
    await g.ensureServing();
    calls.length = 0;
    await g.ensureMining();
    expect(calls).toEqual(['stopLlm', 'startMiner']);
    expect(g.state).toBe(MINING);
  });

  test('never releases while a request is in flight', async () => {
    const { g } = mkGate();
    await g.ensureServing();
    g.begin();
    expect(await g.ensureMining()).toBe(false);
    expect(g.state).toBe(SERVING);
    g.end();
    expect(await g.ensureMining()).toBe(true);
  });

  test('shouldRelease waits for the idle window, and a long generation defers it', async () => {
    let t = 0;
    const { g } = mkGate({ now: () => t, quietMs: 1000 });
    await g.ensureServing();
    t = 500;
    expect(g.shouldRelease()).toBe(false);   // window not elapsed
    t = 2000;
    expect(g.shouldRelease()).toBe(true);
    g.begin();                                // a request arrives
    expect(g.shouldRelease()).toBe(false);   // in flight: the clock cannot evict it
    g.end();
    expect(g.shouldRelease()).toBe(false);   // end() refreshed the activity stamp
    t = 4000;
    expect(g.shouldRelease()).toBe(true);
  });

  test('a failed LLM start leaves the gate mining rather than wedged mid-switch', async () => {
    const { g } = mkGate({ startLlm: async () => { throw new Error('no VRAM'); } });
    await expect(g.ensureServing()).rejects.toThrow('no VRAM');
    expect(g.state).toBe(MINING);
    // and it must be retryable, not permanently stuck holding the transition
    expect(g._transition).toBeNull();
  });
});

describe('LlmGate edges', () => {
  test('constructs with no dependencies at all and still transitions', async () => {
    const g = new LlmGate();
    expect(g.state).toBe(MINING);
    expect(g.quietMs).toBe(60000);
    await g.ensureServing();          // no start/stop callbacks supplied
    expect(g.state).toBe(SERVING);
    await g.ensureMining();
    expect(g.state).toBe(MINING);
  });

  test('setting the state it already has emits nothing', () => {
    const { g } = mkGate();
    const seen = [];
    g.on('state', (s) => seen.push(s));
    g._setState(MINING);
    expect(seen).toEqual([]);
  });

  test('waking when already serving and ready short-circuits', async () => {
    const { g, calls, setReady } = mkGate();
    await g.ensureServing();
    setReady(true);
    calls.length = 0;
    expect(await g.ensureServing()).toBe(true);
    expect(calls).toEqual([]);
  });

  test('releasing when already mining short-circuits', async () => {
    const { g, calls } = mkGate();
    calls.length = 0;
    expect(await g.ensureMining()).toBe(true);
    expect(calls).toEqual([]);
  });

  test('a release racing an in-flight wake joins that transition', async () => {
    const { g } = mkGate();
    const wake = g.ensureServing();
    const release = g.ensureMining();   // arrives mid-switch
    await Promise.all([wake, release]);
    expect(g.state).toBe(SERVING);
  });

  test('quietFor measures from the last activity', () => {
    let t = 100;
    const { g } = mkGate({ now: () => t });
    g.begin(); g.end();
    t = 400;
    expect(g.quietFor()).toBe(300);
  });

  test('end() never drives the in-flight count negative', () => {
    const { g } = mkGate();
    g.end(); g.end();
    expect(g.inFlight).toBe(0);
  });
});

test('the default readiness probe reports not-ready', async () => {
  const g = new LlmGate();
  expect(g.isLlmReady()).toBe(false);
  await g.ensureServing();
  // still uses the default probe, so a second wake re-runs the transition
  expect(await g.ensureServing()).toBe(true);
});
