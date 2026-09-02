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

describe('LlmGate: a failed wake must put the miner back', () => {
  test('startLlm throwing restarts the miner it already stopped', async () => {
    const { g, calls } = mkGate({ startLlm: async () => { throw new Error('no vram'); } });
    await expect(g.ensureServing()).rejects.toThrow('no vram');
    // Without the restart the node ran NEITHER engine: the release timer only
    // fires from SERVING and ensureMining() short-circuits on MINING, so nothing
    // recovered it. It sat at zero while /health still answered.
    expect(calls).toEqual(['stopMiner', 'startMiner']);
    expect(g.state).toBe(MINING);
  });

  test('a miner that also fails to restart still settles in MINING', async () => {
    const { g } = mkGate({
      startLlm: async () => { throw new Error('no vram'); },
      startMiner: async () => { throw new Error('cannot restart either'); },
    });
    // The original error is what the caller needs; the restart is best-effort.
    await expect(g.ensureServing()).rejects.toThrow('no vram');
    expect(g.state).toBe(MINING);
  });

  test('a gate with no startMiner configured does not throw on a failed wake', async () => {
    const g = new LlmGate({
      isLlmReady: () => false,
      startLlm: async () => { throw new Error('no vram'); },
      stopMiner: async () => {},
    });
    await expect(g.ensureServing()).rejects.toThrow('no vram');
    expect(g.state).toBe(MINING);
  });
});

describe('LlmGate: a wake must not join a release', () => {
  test('a request arriving during a release waits, then genuinely wakes', async () => {
    // _transition is one slot shared with ensureMining, so a request arriving
    // during a release used to await the RELEASE, get `true` back, and forward
    // into a model that had just been stopped: a 502 on the HTTP path, and on the
    // cluster path a claimed job reported FAILED. The window is the whole release,
    // which waits for the model's VRAM to come back -- up to 30s.
    const calls = [];
    let releaseStop;
    let ready = true;
    const g = new LlmGate({
      isLlmReady: () => ready,
      startLlm: async () => { calls.push('startLlm'); ready = true; },
      stopLlm: () => { calls.push('stopLlm'); ready = false; return new Promise((r) => { releaseStop = r; }); },
      startMiner: async () => { calls.push('startMiner'); },
      stopMiner: async () => { calls.push('stopMiner'); },
    });
    g.state = SERVING;

    const releasing = g.ensureMining();
    await Promise.resolve();
    const waking = g.ensureServing();       // lands mid-release

    releaseStop();
    await releasing;
    await expect(waking).resolves.toBe(true);

    // The wake actually ran rather than inheriting the release's `true`.
    expect(calls).toEqual(['stopLlm', 'startMiner', 'stopMiner', 'startLlm']);
    expect(g.state).toBe(SERVING);
    expect(g.isLlmReady()).toBe(true);
  });

  test('a failed release still lets the next request wake', async () => {
    let n = 0;
    const g = new LlmGate({
      isLlmReady: () => false,
      stopLlm: async () => { n += 1; if (n === 1) throw new Error('stuck'); },
      startMiner: async () => {},
      stopMiner: async () => {},
      startLlm: async () => {},
    });
    g.state = SERVING;
    const releasing = g.ensureMining().catch(() => {});
    await Promise.resolve();
    const waking = g.ensureServing();
    await releasing;
    await expect(waking).resolves.toBe(true);
  });

  test('concurrent wakes still share one transition', async () => {
    // The direction check must not break the coalescing it was added around.
    const { g, calls } = mkGate();
    const a = g.ensureServing();
    const b = g.ensureServing();
    await Promise.all([a, b]);
    expect(calls).toEqual(['stopMiner', 'startLlm']);
  });
});
